import { Box as LucideBox, LucideProps } from 'lucide-react';

const BoxIcon = ({ className, ...props }: LucideProps) => {
  return <LucideBox className={className} {...props} />;
};

export default BoxIcon;