import { Info as LucideInfo, LucideProps } from 'lucide-react';

const Info = ({ className, ...props }: LucideProps) => {
  return <LucideInfo className={className} {...props} />;
};

export default Info;