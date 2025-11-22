import { Package as LucidePackage, LucideProps } from 'lucide-react';

const BoxLine = ({ className, ...props }: LucideProps) => {
  return <LucidePackage className={className} {...props} />;
};

export default BoxLine;